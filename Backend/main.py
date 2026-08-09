import os
import uuid
from typing import Optional, List, Dict
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from dotenv import load_dotenv
from supabase import create_client, Client
from pydantic import BaseModel

load_dotenv()

# --- Config ---
url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(url, key)

app = FastAPI(title="OFFSIDE: Advanced Sports Backend")

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Security: Bearer Token scheme ---
security = HTTPBearer()


# ============================================================
# --- Auth Guard (JWT verified in FastAPI)
# ============================================================

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    """
    - Backend uses the service key → bypasses Supabase RLS completely.
    - FastAPI manually verifies the JWT token by calling Supabase Auth.
    - If the token is invalid or expired → 401 Unauthorized.
    - If valid → returns the user object so endpoints know who is calling.
    
    Every protected endpoint adds:  user = Depends(get_current_user)
    Public endpoints (login/signup) do NOT use this dependency.
    """
    token = credentials.credentials
    try:
        # Ask Supabase Auth to validate the token and return the user
        user_res = supabase.auth.get_user(token)
        if not user_res or not user_res.user:
            raise HTTPException(status_code=401, detail="Invalid or expired token.")
        return user_res.user
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired token.")


# ============================================================
# --- Schemas (Request/Response Models) ---
# ============================================================

# --- Auth Schemas ---
class SignupRequest(BaseModel):
    email: str
    password: str
    name: str
    nationality: str
    phone_number: str

class LoginRequest(BaseModel):
    email: str
    password: str

# --- Profile Schemas ---
class PlayerProfile(BaseModel):
    full_name: str
    email: str
    phone_number: str
    jersey_number: int
    position: str
    nationality: str
    team_id: Optional[int] = None
    height: float
    weight: float

class UpdatePlayerProfile(BaseModel):
    full_name: Optional[str] = None
    phone_number: Optional[str] = None
    jersey_number: Optional[int] = None
    position: Optional[str] = None
    nationality: Optional[str] = None
    height: Optional[float] = None
    weight: Optional[float] = None

class UpdateUserProfile(BaseModel):
    name: Optional[str] = None
    phone_number: Optional[str] = None
    nationality: Optional[str] = None

# --- Team Schemas ---
class TeamCreate(BaseModel):
    team_name: str
    primary_tshirt_colors: str
    secondary_tshirt_colors: str
    goalkeeper_tshirt_colors: str

# --- Match Schemas ---
class MatchResult(BaseModel):
    match_id: int
    player_id: int
    goals: int
    assists: int
    yellow_card: int
    red_card: int
    is_mvp: bool
    acquisition: float
    top_speed: Optional[float] = None
    total_distance: Optional[float] = None
    actions_detected: Optional[dict] = None
    heatmap_image_url: Optional[str] = None

class TeamMatchStats(BaseModel):
    match_id: int
    team_id: int
    goals: int
    passes: int
    foul: int
    corner: int
    acquisition_avg: float

# --- Transfer & Invitation Schemas ---
class TransferRequest(BaseModel):
    player_id: int
    team_id: int
    tournament_id: int

class InvitationSend(BaseModel):
    team_id: int
    player_id: int

class InvitationAction(BaseModel):
    invitation_id: int

# --- Favourites Schema ---
class FavouriteAdd(BaseModel):
    player_id: int
    entity_type: str  # "player", "team", or "tournament"
    entity_id: int

# --- AI Model Integration Schemas ---
class PlayerAIStat(BaseModel):
    track_id: int
    player_name: Optional[str] = None
    total_distance: float
    top_speed: float

class TeamAIStat(BaseModel):
    possession: Dict[str, float]
    passes_red: int
    passes_green: int
    interceptions_red: int
    interceptions_green: int

class AIMatchResult(BaseModel):
    match_id: int
    team_stats: TeamAIStat
    player_stats: List[PlayerAIStat]
    heatmap_urls: Dict[str, str]


# ============================================================
# --- 1. Authentication (PUBLIC — no token required) ---
# ============================================================

@app.post("/api/v1/auth/signup")
def signup(req: SignupRequest):
    """
    PUBLIC endpoint.
    Creates a Supabase Auth account (handles password hashing automatically),
    then creates the linked USERS record.
    Returns the access_token so Flutter can immediately use the app.
    """
    try:
        # Step 1: Create auth account in Supabase Auth
        auth_res = supabase.auth.sign_up({
            "email": req.email,
            "password": req.password
        })

        if not auth_res.user:
            raise HTTPException(status_code=400, detail="Signup failed.")

        auth_uid = auth_res.user.id  # UUID from Supabase Auth

        # Step 2: Create USERS record linked to auth account
        user_res = supabase.table("USERS").insert({
            "auth_uid":     auth_uid,
            "name":         req.name,
            "email":        req.email,
            "nationality":  req.nationality,
            "phone_number": req.phone_number,
        }).execute()

        return {
            "status": "success",
            "message": "Account created successfully.",
            "auth_uid": auth_uid,
            "user": user_res.data[0],
            "access_token": auth_res.session.access_token if auth_res.session else None,
            "token_type": "bearer",
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Signup failed: {str(e)}")


@app.post("/api/v1/auth/login")
def login(req: LoginRequest):
    """
    PUBLIC endpoint.
    Authenticates with email/password.
    Returns access_token — Flutter stores this and sends it with every request.
    """
    try:
        auth_res = supabase.auth.sign_in_with_password({
            "email": req.email,
            "password": req.password
        })

        if not auth_res.user:
            raise HTTPException(status_code=401, detail="Invalid email or password.")

        # Fetch the USERS record to return user_id to Flutter
        user_res = supabase.table("USERS").select("*")\
            .eq("auth_uid", auth_res.user.id).single().execute()

        return {
            "status": "success",
            "access_token": auth_res.session.access_token,
            "token_type": "bearer",
            "auth_uid": auth_res.user.id,
            "user": user_res.data,
        }
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid email or password.")


@app.post("/api/v1/auth/logout")
def logout(user=Depends(get_current_user)):
    """PROTECTED. Invalidates the current session."""
    try:
        supabase.auth.sign_out()
        return {"status": "success", "message": "Logged out successfully."}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ============================================================
# --- 2. Player Onboarding 
# ============================================================

@app.post("/api/v1/players/profile")
def complete_player_profile(profile: PlayerProfile, user=Depends(get_current_user)):
    """
    PROTECTED.
    Creates an athletic Player profile linked to the logged-in auth account.
    """
    try:
        data = profile.dict()
        data["auth_uid"] = user.id  # Link player to auth account
        res = supabase.table("PLAYERS").insert(data).execute()
        return {"status": "success", "profile": res.data[0]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ============================================================
# --- 3. Team Management 
# ============================================================

@app.post("/api/v1/teams")
def create_team(team: TeamCreate, user=Depends(get_current_user)):
    """PROTECTED. Creates a new team."""
    try:
        res = supabase.table("TEAMS").insert(team.dict()).execute()
        return {"status": "success", "team": res.data[0]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/v1/teams/discover")
def list_teams(user=Depends(get_current_user)):
    """PROTECTED. Returns all teams for the join-team selection screen."""
    try:
        res = supabase.table("TEAMS").select(
            "team_id, team_name, primary_tshirt_colors, "
            "secondary_tshirt_colors, goalkeeper_tshirt_colors"
        ).execute()
        return {"status": "success", "teams": res.data}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/v1/teams/{team_id}")
def get_team_details(team_id: int, user=Depends(get_current_user)):
    """PROTECTED. Returns full details and aggregate stats for a single team."""
    try:
        res = supabase.table("TEAMS").select("*").eq("team_id", team_id).single().execute()
        return {"status": "success", "team": res.data}
    except Exception as e:
        raise HTTPException(status_code=404, detail="Team not found")


# ============================================================
# --- 4. Tournament Management 
# ============================================================

@app.get("/api/v1/tournaments/active")
def get_tournaments(user=Depends(get_current_user)):
    """PROTECTED. Fetches all tournaments and their associated matches."""
    try:
        res = supabase.table("TOURNAMENTS").select("*, MATCHES(*)").execute()
        return {"status": "success", "data": res.data}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ============================================================
# --- 5. Match Flow 
# ============================================================

@app.get("/api/v1/matches/{match_id}")
def get_match_details(match_id: int, user=Depends(get_current_user)):
    """PROTECTED. Returns full match details including team and player stats."""
    try:
        res = supabase.table("MATCHES").select(
            "*, TEAM_MATCH_STATS(*), PLAYER_MATCH_STATS(*)"
        ).eq("match_id", match_id).single().execute()
        return {"status": "success", "match": res.data}
    except Exception as e:
        raise HTTPException(status_code=404, detail="Match not found")


# ============================================================
# --- 6. Match Result Submission 
# ============================================================

@app.post("/api/v1/matches/result")
def submit_match_result(result: MatchResult, user=Depends(get_current_user)):
    """PROTECTED. Submits manual player stats and updates career totals."""
    try:
        supabase.table("PLAYER_MATCH_STATS").insert(result.dict()).execute()

        player_res = supabase.table("PLAYERS").select(
            "total_goals, total_assists, total_yellow_card, total_red_card, "
            "total_appearance, total_acquisition, highest_speed, total_destination"
        ).eq("player_id", result.player_id).single().execute()

        player = player_res.data

        updated_totals = {
            "total_goals":       player["total_goals"] + result.goals,
            "total_assists":     player["total_assists"] + result.assists,
            "total_yellow_card": player["total_yellow_card"] + result.yellow_card,
            "total_red_card":    player["total_red_card"] + result.red_card,
            "total_appearance":  player["total_appearance"] + 1,
            "total_acquisition": round(player["total_acquisition"] + result.acquisition, 2),
        }

        if result.top_speed and result.top_speed > (player["highest_speed"] or 0.0):
            updated_totals["highest_speed"] = result.top_speed

        if result.total_distance:
            updated_totals["total_destination"] = round(
                (player["total_destination"] or 0.0) + result.total_distance, 2
            )

        supabase.table("PLAYERS").update(updated_totals)\
            .eq("player_id", result.player_id).execute()

        return {"status": "success", "message": "Match result submitted and career stats updated."}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/v1/matches/team-stats")
def submit_team_match_stats(stats: TeamMatchStats, user=Depends(get_current_user)):
    """PROTECTED. Submits team-level stats for a match."""
    try:
        supabase.table("TEAM_MATCH_STATS").insert(stats.dict()).execute()

        team_res = supabase.table("TEAMS").select("total_goals, total_passes")\
            .eq("team_id", stats.team_id).single().execute()

        team = team_res.data

        supabase.table("TEAMS").update({
            "total_goals":  team["total_goals"] + stats.goals,
            "total_passes": team["total_passes"] + stats.passes,
        }).eq("team_id", stats.team_id).execute()

        return {"status": "success", "message": "Team match stats submitted."}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ============================================================
# --- 7. Scout Cards & Match History
# ============================================================

@app.get("/api/v1/players/{player_id}/scout-card")
def get_player_stats(player_id: int, user=Depends(get_current_user)):
    """PROTECTED. Returns the full player profile and career stats."""
    try:
        res = supabase.table("PLAYERS").select(
            "full_name, position, jersey_number, nationality, height, weight, "
            "total_goals, total_assists, total_appearance, total_acquisition, "
            "total_yellow_card, total_red_card, highest_speed, total_destination, "
            "email, phone_number, team_id, profile_picture_url"
        ).eq("player_id", player_id).single().execute()
        return {"status": "success", "stats": res.data}
    except Exception as e:
        raise HTTPException(status_code=404, detail="Player not found")


@app.get("/api/v1/players/{player_id}/matches")
def get_player_match_history(player_id: int, user=Depends(get_current_user)):
    """PROTECTED. Returns every match this player has participated in."""
    try:
        res = supabase.table("PLAYER_MATCH_STATS").select(
            "goals, assists, yellow_card, red_card, is_mvp, acquisition, "
            "top_speed, total_distance, heatmap_image_url, actions_detected, "
            "MATCHES(match_date, tournament_id, video_url)"
        ).eq("player_id", player_id).execute()
        return {"status": "success", "history": res.data}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ============================================================
# --- 8. Team Transitions 
# ============================================================

@app.post("/api/v1/players/transfer")
def record_team_transfer(transfer: TransferRequest, user=Depends(get_current_user)):
    """PROTECTED. Updates the player's active team and records the move."""
    try:
        supabase.table("PLAYERS").update({"team_id": transfer.team_id})\
            .eq("player_id", transfer.player_id).execute()

        supabase.table("TEAM_MEMBERS_HISTORY").insert({
            "player_id":     transfer.player_id,
            "team_id":       transfer.team_id,
            "tournament_id": transfer.tournament_id,
        }).execute()

        return {"status": "success", "message": "Transfer recorded successfully."}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/v1/players/{player_id}/team-history")
def get_player_team_history(player_id: int, user=Depends(get_current_user)):
    """PROTECTED. Returns the full team history for a player."""
    try:
        res = supabase.table("TEAM_MEMBERS_HISTORY").select(
            "history_id, "
            "TEAMS(team_name, primary_tshirt_colors, secondary_tshirt_colors, goalkeeper_tshirt_colors), "
            "TOURNAMENTS(tournament_name, start_date, end_date)"
        ).eq("player_id", player_id).execute()
        return {"status": "success", "history": res.data}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ============================================================
# --- 9. Team Invitation System 
# ============================================================

@app.post("/api/v1/invitations/send")
def send_team_invitation(invite: InvitationSend, user=Depends(get_current_user)):
    """PROTECTED. Team manager sends an invitation to a player."""
    try:
        existing = supabase.table("INVITATIONS").select("*")\
            .eq("team_id", invite.team_id)\
            .eq("player_id", invite.player_id)\
            .eq("status", "pending").execute()

        if existing.data:
            return {"status": "error", "message": "An invitation is already pending for this player."}

        res = supabase.table("INVITATIONS").insert(invite.dict()).execute()
        return {"status": "success", "invitation": res.data[0]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/v1/invitations/accept")
def accept_invitation(action: InvitationAction, user=Depends(get_current_user)):
    """PROTECTED. Player accepts an invitation and joins the team."""
    try:
        invite_res = supabase.table("INVITATIONS").select("*")\
            .eq("invitation_id", action.invitation_id).single().execute()

        if not invite_res.data:
            raise HTTPException(status_code=404, detail="Invitation not found")

        invite = invite_res.data

        supabase.table("PLAYERS").update({"team_id": invite["team_id"]})\
            .eq("player_id", invite["player_id"]).execute()

        supabase.table("INVITATIONS").update({"status": "accepted"})\
            .eq("invitation_id", action.invitation_id).execute()

        return {"status": "success", "message": "Player has successfully joined the team."}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/v1/invitations/decline")
def decline_invitation(action: InvitationAction, user=Depends(get_current_user)):
    """PROTECTED. Player declines an invitation."""
    try:
        supabase.table("INVITATIONS").update({"status": "declined"})\
            .eq("invitation_id", action.invitation_id).execute()
        return {"status": "success", "message": "Invitation declined."}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/v1/players/{player_id}/invitations")
def get_my_invitations(player_id: int, user=Depends(get_current_user)):
    """PROTECTED. Returns all pending invitations for a player."""
    try:
        res = supabase.table("INVITATIONS").select("*, TEAMS(team_name)")\
            .eq("player_id", player_id)\
            .eq("status", "pending").execute()
        return {"status": "success", "invitations": res.data}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ============================================================
# --- 10. Profile Picture Upload 
# ============================================================

@app.post("/api/v1/players/{player_id}/upload-photo")
async def upload_profile_picture(
    player_id: int,
    file: UploadFile = File(...),
    user=Depends(get_current_user)
):
    """
    PROTECTED.
    Uploads a player's profile picture to Supabase Storage (avatars bucket)
    and saves the public URL to the PLAYERS table.
    Requires: Supabase Storage bucket named 'avatars' set to Public.
    """
    try:
        file_bytes = await file.read()
        ext = file.filename.split(".")[-1]
        filename = f"player_{player_id}_{uuid.uuid4().hex}.{ext}"

        supabase.storage.from_("avatars").upload(
            path=filename,
            file=file_bytes,
            file_options={"content-type": file.content_type}
        )

        public_url = supabase.storage.from_("avatars").get_public_url(filename)

        supabase.table("PLAYERS").update(
            {"profile_picture_url": public_url}
        ).eq("player_id", player_id).execute()

        return {"status": "success", "profile_picture_url": public_url}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ============================================================
# --- 11. Edit Profile 
# ============================================================

@app.patch("/api/v1/players/{player_id}")
def update_player_profile(
    player_id: int,
    updates: UpdatePlayerProfile,
    user=Depends(get_current_user)
):
    """
    PROTECTED.
    Updates only the fields that are provided — leaves all others unchanged.
    Flutter sends only the fields the user actually edited.
    """
    try:
        data = {k: v for k, v in updates.dict().items() if v is not None}
        if not data:
            raise HTTPException(status_code=400, detail="No fields to update.")

        res = supabase.table("PLAYERS").update(data)\
            .eq("player_id", player_id).execute()
        return {"status": "success", "updated": res.data[0]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.patch("/api/v1/users/{user_id}")
def update_user_profile(
    user_id: int,
    updates: UpdateUserProfile,
    user=Depends(get_current_user)
):
    """PROTECTED. Updates user account info — only provided fields are changed."""
    try:
        data = {k: v for k, v in updates.dict().items() if v is not None}
        if not data:
            raise HTTPException(status_code=400, detail="No fields to update.")

        res = supabase.table("USERS").update(data)\
            .eq("user_id", user_id).execute()
        return {"status": "success", "updated": res.data[0]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ============================================================
# --- 12. Favourites System 
# ============================================================

@app.post("/api/v1/favourites")
def add_favourite(fav: FavouriteAdd, user=Depends(get_current_user)):
    """PROTECTED. Adds a player, team, or tournament to favourites."""
    try:
        res = supabase.table("FAVOURITES").insert(fav.dict()).execute()
        return {"status": "success", "favourite": res.data[0]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.delete("/api/v1/favourites")
def remove_favourite(fav: FavouriteAdd, user=Depends(get_current_user)):
    """PROTECTED. Removes an item from favourites."""
    try:
        supabase.table("FAVOURITES").delete()\
            .eq("player_id", fav.player_id)\
            .eq("entity_type", fav.entity_type)\
            .eq("entity_id", fav.entity_id).execute()
        return {"status": "success", "message": "Removed from favourites."}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/v1/players/{player_id}/favourites")
def get_my_favourites(player_id: int, user=Depends(get_current_user)):
    """
    PROTECTED.
    Returns all favourites grouped by type.
    Response: {"player": [1,2], "team": [3], "tournament": [1]}
    """
    try:
        res = supabase.table("FAVOURITES").select("*")\
            .eq("player_id", player_id).execute()

        grouped: Dict[str, List[int]] = {"player": [], "team": [], "tournament": []}
        for fav in res.data:
            entity_type = fav["entity_type"]
            if entity_type in grouped:
                grouped[entity_type].append(fav["entity_id"])

        return {"status": "success", "favourites": grouped}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/v1/players/{player_id}/favourites/{entity_type}/{entity_id}")
def check_is_favourite(
    player_id: int,
    entity_type: str,
    entity_id: int,
    user=Depends(get_current_user)
):
    """PROTECTED. Checks if an item is favourited. Used for heart icon state in Flutter."""
    try:
        res = supabase.table("FAVOURITES").select("favourite_id")\
            .eq("player_id", player_id)\
            .eq("entity_type", entity_type)\
            .eq("entity_id", entity_id).execute()
        return {"status": "success", "is_favourite": len(res.data) > 0}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ============================================================
# --- 13. AI Model Integration (NO auth — called by pipeline) ---
# ============================================================

@app.post("/api/v1/ai/analyze-match/{match_id}")
def receive_ai_results(match_id: int, result: AIMatchResult):
    """
    NO AUTH — This endpoint is called by the AI pipeline running on the
    teammate's local machine, not by Flutter. Adding auth here would
    require the pipeline to manage tokens which is unnecessary complexity.
    """
    try:
        match_res = supabase.table("MATCHES").select("home_team_id, away_team_id")\
            .eq("match_id", match_id).single().execute()

        home_team_id = match_res.data["home_team_id"]
        away_team_id = match_res.data["away_team_id"]

        supabase.table("TEAM_MATCH_STATS").insert({
            "match_id":        match_id,
            "team_id":         home_team_id,
            "goals":           0,
            "passes":          result.team_stats.passes_red,
            "foul":            0,
            "corner":          0,
            "acquisition_avg": result.team_stats.possession.get("Red Team", 0.0)
        }).execute()

        supabase.table("TEAM_MATCH_STATS").insert({
            "match_id":        match_id,
            "team_id":         away_team_id,
            "goals":           0,
            "passes":          result.team_stats.passes_green,
            "foul":            0,
            "corner":          0,
            "acquisition_avg": result.team_stats.possession.get("Green Team", 0.0)
        }).execute()

        for p_ai in result.player_stats:
            if not p_ai.player_name or p_ai.player_name in ("Unknown", "Identifying..."):
                continue

            player_res = supabase.table("PLAYERS").select("player_id").ilike(
                "full_name", f"%{p_ai.player_name}%"
            ).execute()

            if not player_res.data:
                continue

            player_id = player_res.data[0]["player_id"]
            heatmap_url = result.heatmap_urls.get(p_ai.player_name)

            supabase.table("PLAYER_MATCH_STATS").insert({
                "match_id":       match_id,
                "player_id":      player_id,
                "goals":          0,
                "assists":        0,
                "yellow_card":    0,
                "red_card":       0,
                "is_mvp":         False,
                "acquisition":    0.0,
                "top_speed":      p_ai.top_speed,
                "total_distance": p_ai.total_distance,
                "heatmap_image_url": heatmap_url
            }).execute()

            career_res = supabase.table("PLAYERS").select(
                "highest_speed, total_destination"
            ).eq("player_id", player_id).single().execute()

            career = career_res.data
            updated_career = {}

            if p_ai.top_speed > (career["highest_speed"] or 0.0):
                updated_career["highest_speed"] = p_ai.top_speed

            if p_ai.total_distance:
                updated_career["total_destination"] = round(
                    (career["total_destination"] or 0.0) + p_ai.total_distance, 2
                )

            if updated_career:
                supabase.table("PLAYERS").update(updated_career)\
                    .eq("player_id", player_id).execute()

        return {
            "status": "success",
            "message": f"AI results for match {match_id} saved successfully.",
            "players_mapped": len(result.player_stats)
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
